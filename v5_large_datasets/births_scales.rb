require 'csv'
require 'json'

filenames = [
  "linkco2007us_den.csv",
  "linkco2008us_den.csv",
  "linkco2009us_den.csv",
  "linkco2010us_den.csv",
]

births_columns_numbers = {}
CSV.foreach('births_columns_numbers.csv') do |row|
  column_name, columns_numbers = row
  births_columns_numbers[column_name] = JSON.parse(columns_numbers)
end

births_columns_categories = {}
CSV.foreach('births_columns_categories.csv') do |row|
  column_name, column_categories = row
  births_columns_categories[column_name] = JSON.parse(column_categories)
end

births_columns_nans = {}
CSV.foreach('births_columns_nans.csv') do |row|
  column_name, column_nan = row
  births_columns_nans[column_name] = JSON.parse(column_nan)
end

csv_out = CSV.open('births_scales.csv', 'w')

filenames.each do |filename|
  linenumber = -1
  all_columns = []
  columns = []

  CSV.foreach(filename) do |values|
    linenumber += 1
    puts linenumber

    puts "Processing filename=#{filename},linenumber=#{linenumber}" if linenumber % 10_000 == 0

    if linenumber == 0
      all_columns = values

      # ignore nan column
      if columns.empty?
        columns = all_columns - births_columns_nans.keys
        puts columns.size
        csv_out << columns
      end

      next
    end

    scaled_values_hash = {}
    nan_count = 0
    values.each_with_index do |value, index|
      column_name = all_columns[index]

      # ignore nan column
      scaled_value = if births_columns_nans.keys.include?(column_name)
        nan_count += 1
        nil
      end

      if births_columns_numbers[column_name]
        column_mumbers = births_columns_numbers[column_name]
        # mean is the default vaule for nan
        scaled_value = column_mumbers['mean']

        if value != ''
          min = column_mumbers['min']
          max = column_mumbers['max']
          scaled_value = (value.to_f - min) / (max - min)
        end
      end

      if births_columns_categories[column_name]
        columns_categories = births_columns_categories[column_name]
        scaled_value = columns_categories[value || '']
      end

      # puts "column_name=#{column_name}, value=#{value}, scaled_value=#{scaled_value || 'empty'}"
      # puts "numbers=#{!births_columns_numbers[column_name].nil?}, categories=#{!births_columns_categories[column_name].nil?}"

      scaled_values_hash[column_name] = scaled_value
    end

    # ignore nan column
    scaled_values_hash = scaled_values_hash.select do |k, v|
      !v.nil?
    end

    scaled_values = scaled_values_hash.sort_by do |key, value|
      columns.index(key)
    end.map do |key, value|
      value
    end

    if scaled_values.size == columns.size
      csv_out << scaled_values
    else
      raise "Something wrong index=#{index} scaled_values.size #{scaled_values.size}, columns.size #{columns.size}, #{nan_count}"
    end
  end
end
