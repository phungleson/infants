require 'csv'
require 'json'
require 'finishing_moves'

scaled_values = {}

CSV.foreach('infants_columns_hash.csv') do |row|
  column_name, values_hash = row
  values_hash = JSON.parse(values_hash)
  unique_values = values_hash.keys

  values = unique_values.select { |v| v != '' }

  if values.size > 0 && values[0].numeric?
    values = values.map(&:to_f)
    values_count = values_hash.values.reduce(&:+)
    values_sum = values_hash.map do |key, value|
      key.to_f * value
    end.reduce(&:+)

    scaled_values[column_name] ||= {}
    scaled_values[column_name][:mean] = values_sum / values_count
  end
end

CSV.foreach('infants_columns.csv') do |row|
  column_name, unique_values = row
  unique_values = JSON.parse(unique_values)

  values = unique_values.select { |v| v != nil }

  if values.size > 0 && values[0].numeric?
    unique_values = unique_values.map(&:to_f)

    scaled_values[column_name] ||= {}
    scaled_values[column_name][:min] = unique_values.min
    scaled_values[column_name][:max] = unique_values.max
  end
end

csv = CSV.open('infants_columns_stats.csv', 'wb')

scaled_values.each do |k, values|
  csv << [k, values.to_json]
end